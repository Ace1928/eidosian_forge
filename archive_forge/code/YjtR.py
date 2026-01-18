from PIL import Image, ImageDraw, ImageFont
import os

# Directory containing the images
image_dir = "/path/to/image/directory"

# Desired size for resized images
new_size = (800, 600)

# Watermark text and font
watermark_text = "Your Watermark"
font = ImageFont.truetype("arial.ttf", 36)

# Iterate over the images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)

        # Resize the image
        resized_image = image.resize(new_size)

        # Add watermark
        draw = ImageDraw.Draw(resized_image)
        text_width, text_height = draw.textsize(watermark_text, font)
        x = resized_image.width - text_width - 10
        y = resized_image.height - text_height - 10
        draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 128))

        # Save the modified image
        new_filename = f"resized_{filename}"
        new_image_path = os.path.join(image_dir, new_filename)
        resized_image.save(new_image_path)
